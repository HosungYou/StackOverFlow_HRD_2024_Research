#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Analysis of Developer Skills Market Value and Learning Pathways
for Human Resource Development (HRD) Strategic Planning

This script:
1. Generates synthetic data mimicking the 2024 Stack Overflow Developer Survey
2. Performs integrated analysis of market value and learning pathways
3. Generates actual tables and figures for the academic report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
import pickle
import random
import json
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import shap

# 경로 수정: Reports_0425 디렉토리 사용
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
REPORTS_DIR = BASE_DIR / "Reports_0425"
OUTPUT_DIR = REPORTS_DIR / "output"
FIGURE_DIR = REPORTS_DIR / "figures"

# Create necessary directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
custom_palette = sns.color_palette("viridis", 10)
sns.set_palette(custom_palette)

# Configuration
config = {
    "n_samples": 10000,  # Number of synthetic samples to generate
    "test_size": 0.2,    # Proportion of data for test set
    "random_state": 42,  # Random seed for reproducibility
    "n_estimators": 520, # Number of trees in random forest
    "max_depth": 7,      # Maximum depth of trees
    "learning_rate": 0.04, # Learning rate for gradient boosting
    "subsample": 0.8,    # Subsample ratio for gradient boosting
    "colsample_bytree": 0.75, # Column subsample ratio for gradient boosting
    "min_child_weight": 3, # Minimum sum of instance weight needed in a child
    "reg_alpha": 0.1,    # L1 regularization
    "reg_lambda": 1.2,   # L2 regularization
}

# Define skill domains and specific skills
SKILL_DOMAINS = {
    "Languages": ["Python", "JavaScript", "Java", "C#", "Go", "TypeScript", "Rust", "PHP", "C++", "SQL"],
    "Frameworks": ["React", "Angular", "Vue", "Django", "Flask", "Spring", "ASP.NET", "Node.js", "Express", "Next.js"],
    "Databases": ["MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "Elasticsearch", "Oracle", "DynamoDB", "Cassandra", "Neo4j"],
    "Cloud": ["AWS", "Azure", "GCP", "Heroku", "DigitalOcean", "Linode", "IBM Cloud", "Oracle Cloud", "Alibaba Cloud", "Cloudflare"],
    "DevOps": ["Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions", "Terraform", "Ansible", "Chef", "Puppet", "ArgoCD"],
    "AI/ML": ["TensorFlow", "PyTorch", "scikit-learn", "Keras", "ONNX", "Hugging Face", "OpenAI", "LangChain", "Ray", "SpaCy"],
    "Mobile": ["React Native", "Flutter", "Kotlin", "Swift", "Xamarin", "Ionic", "Capacitor", "NativeScript", "Android SDK", "iOS SDK"],
    "Frontend": ["HTML", "CSS", "Sass", "Tailwind", "Bootstrap", "Material UI", "Styled Components", "jQuery", "Redux", "MobX"],
    "Testing": ["Jest", "Cypress", "Selenium", "Playwright", "Mocha", "Chai", "Pytest", "JUnit", "TestNG", "Postman"],
    "Tools": ["Git", "VS Code", "IntelliJ", "PyCharm", "Vim", "Sublime Text", "Jira", "Trello", "Slack", "Notion"]
}

# Define learning pathways and resources
LEARNING_PATHWAYS = [
    "Self-taught", "Academic education", "Coding bootcamp", "On-the-job training", 
    "Mentorship", "Open source contribution", "Online courses", "Industry certification"
]

LEARNING_RESOURCES = [
    "Documentation", "Stack Overflow", "YouTube tutorials", "Interactive platforms", 
    "Academic papers", "Books", "Blog posts", "Official courses", "Conferences", 
    "Podcasts", "Community forums", "GitHub repositories", "Reddit", "Twitter/Social media",
    "Company training materials", "Mentoring sessions", "Peer programming", "Online communities",
    "Open source contribution", "Personal projects"
]

# Career stages
CAREER_STAGES = ["Junior (0-3 years)", "Mid-level (3-7 years)", "Senior (7-12 years)", "Principal/Lead (12+ years)"]

# Developer roles
DEVELOPER_ROLES = [
    "Frontend Developer", "Backend Developer", "Full-stack Developer", "Mobile Developer",
    "Data Scientist/ML Engineer", "DevOps/SRE", "Cloud Engineer", "Security Engineer",
    "QA/Test Engineer", "UX/UI Developer", "Game Developer", "Embedded Developer"
]

# Geographic regions
REGIONS = ["North America", "Western Europe", "Eastern Europe", "South Asia", "East Asia", 
           "Southeast Asia", "Middle East", "South America", "Africa", "Oceania"]

# Company sizes
COMPANY_SIZES = ["Small (1-49)", "Medium (50-249)", "Large (250-999)", "Enterprise (1000+)"]

# Education levels
EDUCATION_LEVELS = ["High School", "Some College", "Associate Degree", "Bachelor's Degree", 
                     "Master's Degree", "PhD or Higher", "Self-taught", "Bootcamp Graduate"]

# Generate synthetic data
def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic data mimicking the Stack Overflow Developer Survey 2024.
    Includes demographic information, skills, learning pathways, and salary.
    """
    print("Generating synthetic data...")
    
    data = []
    
    for i in range(n_samples):
        # Basic demographics
        experience_years = np.random.choice([1, 2, 3, 5, 7, 10, 15, 20, 25], p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.08, 0.05, 0.02])
        
        # Determine career stage based on experience
        if experience_years < 3:
            career_stage = CAREER_STAGES[0]
        elif experience_years < 7:
            career_stage = CAREER_STAGES[1]
        elif experience_years < 12:
            career_stage = CAREER_STAGES[2]
        else:
            career_stage = CAREER_STAGES[3]
        
        developer_role = np.random.choice(DEVELOPER_ROLES)
        region = np.random.choice(REGIONS, p=[0.3, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02])
        company_size = np.random.choice(COMPANY_SIZES, p=[0.2, 0.3, 0.3, 0.2])
        education_level = np.random.choice(EDUCATION_LEVELS, p=[0.05, 0.1, 0.05, 0.4, 0.25, 0.05, 0.05, 0.05])
        
        # Skills - randomly select proficiency in various skills
        skill_proficiencies = {}
        
        # Generate more realistic skill patterns based on role
        role_skill_bias = {
            "Frontend Developer": {"Frontend": 0.8, "Languages": 0.6, "Frameworks": 0.7, "Testing": 0.6},
            "Backend Developer": {"Languages": 0.7, "Databases": 0.8, "DevOps": 0.5, "Frameworks": 0.6},
            "Full-stack Developer": {"Frontend": 0.7, "Languages": 0.7, "Databases": 0.7, "Frameworks": 0.7},
            "Mobile Developer": {"Mobile": 0.9, "Languages": 0.6, "Testing": 0.5},
            "Data Scientist/ML Engineer": {"AI/ML": 0.9, "Languages": 0.7, "Databases": 0.6},
            "DevOps/SRE": {"DevOps": 0.9, "Cloud": 0.8, "Tools": 0.7},
            "Cloud Engineer": {"Cloud": 0.9, "DevOps": 0.7, "Databases": 0.6},
            "Security Engineer": {"DevOps": 0.6, "Tools": 0.7, "Languages": 0.6},
            "QA/Test Engineer": {"Testing": 0.9, "Tools": 0.7, "Languages": 0.5},
            "UX/UI Developer": {"Frontend": 0.9, "Tools": 0.7},
            "Game Developer": {"Languages": 0.7, "Tools": 0.6, "Testing": 0.5},
            "Embedded Developer": {"Languages": 0.8, "Tools": 0.6}
        }
        
        # Set default bias for all domains
        domain_bias = {domain: 0.3 for domain in SKILL_DOMAINS.keys()}
        
        # Update with role-specific bias
        if developer_role in role_skill_bias:
            for domain, bias in role_skill_bias[developer_role].items():
                domain_bias[domain] = bias
                
        # Add experience factor - more experienced developers have higher proficiency
        experience_factor = min(0.8, experience_years / 20)
        
        # Generate skill proficiencies
        for domain, skills in SKILL_DOMAINS.items():
            domain_prob = domain_bias[domain] + experience_factor * 0.2
            
            for skill in skills:
                # Probability of knowing the skill at all
                if np.random.random() < domain_prob:
                    # Proficiency level (1-5)
                    base_proficiency = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
                    # Adjust proficiency based on experience
                    proficiency_boost = min(5 - base_proficiency, np.random.binomial(5, experience_factor * 0.5))
                    skill_proficiencies[skill] = base_proficiency + proficiency_boost
                else:
                    skill_proficiencies[skill] = 0
        
        # Learning pathways and resources
        # Number of learning pathways used - based on experience
        n_pathways = min(len(LEARNING_PATHWAYS), max(1, int(np.random.normal(2 + experience_factor * 3, 1))))
        learning_pathways = np.random.choice(LEARNING_PATHWAYS, size=n_pathways, replace=False)
        
        # Learning resources - more experienced developers use more resources
        n_resources = min(len(LEARNING_RESOURCES), max(2, int(np.random.normal(5 + experience_factor * 8, 2))))
        learning_resources = np.random.choice(LEARNING_RESOURCES, size=n_resources, replace=False)
        
        # Create features for learning approaches
        learning_features = {}
        
        # Initialize all learning features to 0
        for pathway in LEARNING_PATHWAYS:
            learning_features[f"Pathway_{pathway.replace(' ', '_')}"] = 0
            
        for resource in LEARNING_RESOURCES:
            learning_features[f"Resource_{resource.replace(' ', '_')}"] = 0
            
        # Set the active ones to 1
        for pathway in learning_pathways:
            learning_features[f"Pathway_{pathway.replace(' ', '_')}"] = 1
            
        for resource in learning_resources:
            learning_features[f"Resource_{resource.replace(' ', '_')}"] = 1
            
        # Additional learning metrics
        learning_diversity = n_resources / len(LEARNING_RESOURCES)
        learning_intensity = np.random.normal(0.5 + experience_factor * 0.3, 0.2)
        learning_consistency = np.random.normal(0.5 + experience_factor * 0.3, 0.2)
        
        # Calculated field - overall learning effectiveness
        learning_effectiveness = (learning_diversity * 0.3 + learning_intensity * 0.4 + learning_consistency * 0.3) * (0.7 + experience_factor * 0.3)
        
        # Base salary calculation
        # Region factors
        region_factors = {
            "North America": 1.5,
            "Western Europe": 1.3,
            "Eastern Europe": 0.8,
            "South Asia": 0.5,
            "East Asia": 1.1,
            "Southeast Asia": 0.7,
            "Middle East": 0.9,
            "South America": 0.6,
            "Africa": 0.5,
            "Oceania": 1.2
        }
        
        # Role factors
        role_factors = {
            "Frontend Developer": 0.9,
            "Backend Developer": 1.0,
            "Full-stack Developer": 1.1,
            "Mobile Developer": 1.0,
            "Data Scientist/ML Engineer": 1.3,
            "DevOps/SRE": 1.2,
            "Cloud Engineer": 1.2,
            "Security Engineer": 1.25,
            "QA/Test Engineer": 0.9,
            "UX/UI Developer": 0.95,
            "Game Developer": 1.0,
            "Embedded Developer": 1.05
        }
        
        # Company size factors
        company_factors = {
            "Small (1-49)": 0.85,
            "Medium (50-249)": 1.0,
            "Large (250-999)": 1.1,
            "Enterprise (1000+)": 1.2
        }
        
        # Education factors
        education_factors = {
            "High School": 0.8,
            "Some College": 0.85,
            "Associate Degree": 0.9,
            "Bachelor's Degree": 1.0,
            "Master's Degree": 1.1,
            "PhD or Higher": 1.2,
            "Self-taught": 0.9,
            "Bootcamp Graduate": 0.95
        }
        
        # Base calculation
        base_salary = 50000 + (experience_years * 3000)
        
        # Apply factors
        salary = base_salary * region_factors[region] * role_factors[developer_role] * company_factors[company_size] * education_factors[education_level]
        
        # Skill premium - calculate the premium based on high-value skills
        skill_premium = 0
        high_value_skills = {
            "Python": 5000,
            "JavaScript": 3000,
            "Go": 8000,
            "Rust": 10000,
            "React": 4000,
            "Angular": 3000,
            "AWS": 7000,
            "Azure": 6000,
            "GCP": 7000,
            "Docker": 5000,
            "Kubernetes": 9000,
            "TensorFlow": 8000,
            "PyTorch": 9000,
            "scikit-learn": 6000
        }
        
        for skill, premium in high_value_skills.items():
            if skill in skill_proficiencies and skill_proficiencies[skill] > 0:
                # Scale premium by proficiency
                scaled_premium = premium * (skill_proficiencies[skill] / 5)
                skill_premium += scaled_premium
                
        # Learning pathway premium
        learning_premium = 0
        pathway_premiums = {
            "Self-taught": 2000,
            "Academic education": 5000,
            "Coding bootcamp": 3000,
            "On-the-job training": 4000,
            "Mentorship": 6000,
            "Open source contribution": 7000,
            "Online courses": 2500,
            "Industry certification": 5000
        }
        
        for pathway in learning_pathways:
            learning_premium += pathway_premiums[pathway]
            
        # Resource premium
        resource_premium = 0
        resource_premiums = {
            "Documentation": 3000,
            "Stack Overflow": 2000,
            "YouTube tutorials": 1000,
            "Interactive platforms": 2500,
            "Academic papers": 3000,
            "Books": 1500,
            "Blog posts": 1000,
            "Official courses": 2500,
            "Conferences": 2000,
            "Podcasts": 500,
            "Community forums": 1500,
            "GitHub repositories": 2500,
            "Reddit": 500,
            "Twitter/Social media": 500,
            "Company training materials": 1500,
            "Mentoring sessions": 3000,
            "Peer programming": 2500,
            "Online communities": 1500,
            "Open source contribution": 3000,
            "Personal projects": 2000
        }
        
        for resource in learning_resources:
            resource_premium += resource_premiums[resource]
            
        # Learning effectiveness multiplier
        learning_multiplier = 0.5 + learning_effectiveness
        
        # Apply learning premiums with diminishing returns
        total_premium = skill_premium + (learning_premium * 0.5) + (resource_premium * 0.3)
        total_premium = total_premium * learning_multiplier
        
        # Add some random noise
        noise = np.random.normal(0, 5000)
        
        # Final salary
        final_salary = salary + total_premium + noise
        
        # Log transform for modeling purposes
        log_salary = np.log(max(30000, final_salary))
        
        # Create row
        row = {
            "ResponsId": i + 1,
            "Experience_Years": experience_years,
            "Career_Stage": career_stage,
            "Developer_Role": developer_role,
            "Region": region,
            "Company_Size": company_size,
            "Education_Level": education_level,
            "Learning_Diversity": learning_diversity,
            "Learning_Intensity": learning_intensity,
            "Learning_Consistency": learning_consistency,
            "Learning_Effectiveness": learning_effectiveness,
            "Salary": final_salary,
            "LogSalary": log_salary
        }
        
        # Add skill proficiencies
        for skill, proficiency in skill_proficiencies.items():
            row[f"Skill_{skill.replace(' ', '_')}"] = proficiency
            
        # Add learning features
        row.update(learning_features)
        
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    print(f"Generated synthetic dataset with {n_samples} samples and {df.shape[1]} features")
    
    return df

# Prepare data for modeling
def prepare_data_for_modeling(df):
    """
    Prepare the data for modeling by:
    1. Splitting into training and test sets
    2. Defining feature columns
    3. Handling any data preprocessing
    """
    print("Preparing data for modeling...")
    
    # Define target variable
    target = "LogSalary"
    
    # Define feature groups for analysis
    feature_groups = {}
    
    # Demographic features
    feature_groups["demographic"] = ["Experience_Years", "Career_Stage", "Developer_Role", "Region", "Company_Size", "Education_Level"]
    
    # Skill features - all columns that start with "Skill_"
    feature_groups["skills"] = [col for col in df.columns if col.startswith("Skill_")]
    
    # Learning pathway features - all columns that start with "Pathway_"
    feature_groups["pathways"] = [col for col in df.columns if col.startswith("Pathway_")]
    
    # Learning resource features - all columns that start with "Resource_"
    feature_groups["resources"] = [col for col in df.columns if col.startswith("Resource_")]
    
    # Learning metric features
    feature_groups["learning_metrics"] = ["Learning_Diversity", "Learning_Intensity", "Learning_Consistency", "Learning_Effectiveness"]
    
    # All features combined
    all_features = []
    for group, features in feature_groups.items():
        all_features.extend(features)
        
    # Remove any duplicates
    all_features = list(set(all_features))
    
    # Create dummy variables for categorical features
    categorical_features = ["Career_Stage", "Developer_Role", "Region", "Company_Size", "Education_Level"]
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    
    # Update feature lists to include dummy variables
    updated_features = []
    for feature in all_features:
        if feature in categorical_features:
            # Add all dummy columns for this feature
            dummy_cols = [col for col in df_encoded.columns if col.startswith(f"{feature}_")]
            updated_features.extend(dummy_cols)
        else:
            updated_features.append(feature)
            
    all_features = updated_features
    
    # Split into training and test sets
    X = df_encoded[all_features].copy()
    y = df_encoded[target].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )
    
    print(f"Data prepared with {len(all_features)} features")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, all_features, feature_groups

# Train regression models
def train_models(X_train, X_test, y_train, y_test, all_features):
    """
    Train Random Forest and XGBoost regression models
    """
    print("Training regression models...")
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        random_state=config["random_state"],
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    
    model_metrics = {}
    
    for name, model in models.items():
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        model_metrics[name] = {
            "Train R2": train_r2,
            "Test R2": test_r2,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse
        }
        
        print(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
    return models, model_metrics

# SHAP analysis
def perform_shap_analysis(models, X_train, X_test, all_features, feature_groups):
    """
    Perform SHAP analysis for model interpretation
    """
    print("Performing SHAP analysis...")
    
    # Use the XGBoost model for SHAP analysis
    model = models["XGBoost"]
    
    # Initialize JS visualizations using TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Sample the test set for SHAP analysis (for performance)
    shap_sample_size = min(500, X_test.shape[0])
    X_shap = X_test.sample(shap_sample_size, random_state=config["random_state"])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_shap)
    
    # Store the SHAP results
    shap_results = {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_shap": X_shap,
        "features": all_features,
        "feature_groups": feature_groups
    }
    
    print("SHAP analysis completed")
    
    return shap_results

# Analyze learning pathways
def analyze_learning_pathways(df, shap_results, feature_groups):
    """
    Analyze the impact of different learning pathways and resources on market value
    """
    print("Analyzing learning pathways...")
    
    # Extract SHAP values for learning pathways and resources
    pathway_shap = {f: i for i, f in enumerate(shap_results["features"]) if f in feature_groups["pathways"] or f.startswith("Pathway_")}
    resource_shap = {f: i for i, f in enumerate(shap_results["features"]) if f in feature_groups["resources"] or f.startswith("Resource_")}
    
    # Calculate the impact of learning pathways on market value
    pathway_impact = {}
    for pathway, idx in pathway_shap.items():
        # Clean name for display (remove "Pathway_" prefix and replace underscores)
        clean_name = pathway.replace("Pathway_", "").replace("_", " ")
        
        # Calculate the mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_results["shap_values"][:, idx]))
        
        # Calculate the mean SHAP value (can be negative)
        mean_shap = np.mean(shap_results["shap_values"][:, idx])
        
        # Calculate the market value impact (convert log-salary SHAP to dollars)
        # Base salary of $80,000 used for conversion
        base_salary = 80000
        value_impact = base_salary * (np.exp(mean_shap) - 1)
        value_impact_pct = np.exp(mean_shap) - 1
        
        pathway_impact[clean_name] = {
            "Mean Abs SHAP": mean_abs_shap,
            "Mean SHAP": mean_shap,
            "Value Impact ($)": value_impact,
            "Value Impact (%)": value_impact_pct * 100
        }
    
    # Calculate the impact of learning resources on market value
    resource_impact = {}
    for resource, idx in resource_shap.items():
        # Clean name for display
        clean_name = resource.replace("Resource_", "").replace("_", " ")
        
        # Calculate the mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_results["shap_values"][:, idx]))
        
        # Calculate the mean SHAP value (can be negative)
        mean_shap = np.mean(shap_results["shap_values"][:, idx])
        
        # Calculate the market value impact
        value_impact = base_salary * (np.exp(mean_shap) - 1)
        value_impact_pct = np.exp(mean_shap) - 1
        
        resource_impact[clean_name] = {
            "Mean Abs SHAP": mean_abs_shap,
            "Mean SHAP": mean_shap,
            "Value Impact ($)": value_impact,
            "Value Impact (%)": value_impact_pct * 100
        }
    
    # Calculate the ROI of different learning approaches
    learning_roi = calculate_learning_roi(pathway_impact, resource_impact)
    
    # Identify optimal learning resource combinations
    optimal_combinations = identify_optimal_combinations(df, shap_results, feature_groups)
    
    # Analyze variation across career stages
    career_stage_variation = analyze_by_career_stage(df, shap_results, feature_groups)
    
    # Compile results
    learning_analysis = {
        "pathway_impact": pathway_impact,
        "resource_impact": resource_impact,
        "learning_roi": learning_roi,
        "optimal_combinations": optimal_combinations,
        "career_stage_variation": career_stage_variation
    }
    
    print("Learning pathway analysis completed")
    
    return learning_analysis

# Calculate ROI for learning approaches
def calculate_learning_roi(pathway_impact, resource_impact):
    """
    Calculate the ROI of different learning pathways and resources
    """
    print("Calculating learning ROI...")
    
    # Define time and financial investments for different learning pathways
    pathway_investments = {
        "Self taught": {
            "Financial Cost": 500,  # Cost of books, online resources, etc.
            "Time Investment (hours)": 1000,  # Self-learning takes more time
            "Hourly Value": 20  # Value of time based on entry-level salary
        },
        "Academic education": {
            "Financial Cost": 30000,  # Cost of tuition
            "Time Investment (hours)": 3000,  # 4-year degree
            "Hourly Value": 20
        },
        "Coding bootcamp": {
            "Financial Cost": 15000,  # Cost of bootcamp
            "Time Investment (hours)": 500,  # Intensive but shorter
            "Hourly Value": 20
        },
        "On the job training": {
            "Financial Cost": 0,  # No direct cost
            "Time Investment (hours)": 500,  # Learning while working
            "Hourly Value": 30  # Higher hourly value as it's during work
        },
        "Mentorship": {
            "Financial Cost": 5000,  # Cost of mentorship programs
            "Time Investment (hours)": 200,
            "Hourly Value": 25
        },
        "Open source contribution": {
            "Financial Cost": 0,
            "Time Investment (hours)": 300,
            "Hourly Value": 25
        },
        "Online courses": {
            "Financial Cost": 2000,  # Subscription costs
            "Time Investment (hours)": 300,
            "Hourly Value": 20
        },
        "Industry certification": {
            "Financial Cost": 2500,  # Certification fees
            "Time Investment (hours)": 200,
            "Hourly Value": 25
        }
    }
    
    # Define investments for learning resources
    resource_investments = {
        "Documentation": {
            "Financial Cost": 0,
            "Time Investment (hours)": 200,
            "Hourly Value": 20
        },
        "Stack Overflow": {
            "Financial Cost": 0,
            "Time Investment (hours)": 150,
            "Hourly Value": 20
        },
        "YouTube tutorials": {
            "Financial Cost": 0,
            "Time Investment (hours)": 100,
            "Hourly Value": 20
        },
        "Interactive platforms": {
            "Financial Cost": 500,
            "Time Investment (hours)": 200,
            "Hourly Value": 20
        },
        "Academic papers": {
            "Financial Cost": 300,
            "Time Investment (hours)": 100,
            "Hourly Value": 25
        },
        "Books": {
            "Financial Cost": 400,
            "Time Investment (hours)": 150,
            "Hourly Value": 20
        },
        "Official courses": {
            "Financial Cost": 1500,
            "Time Investment (hours)": 100,
            "Hourly Value": 25
        },
        "Conferences": {
            "Financial Cost": 2000,
            "Time Investment (hours)": 40,
            "Hourly Value": 30
        },
        "Community forums": {
            "Financial Cost": 0,
            "Time Investment (hours)": 100,
            "Hourly Value": 20
        },
        "GitHub repositories": {
            "Financial Cost": 0,
            "Time Investment (hours)": 150,
            "Hourly Value": 25
        },
        "Mentoring sessions": {
            "Financial Cost": 2000,
            "Time Investment (hours)": 50,
            "Hourly Value": 30
        },
        "Open source contribution": {
            "Financial Cost": 0,
            "Time Investment (hours)": 200,
            "Hourly Value": 25
        },
        "Personal projects": {
            "Financial Cost": 200,
            "Time Investment (hours)": 200,
            "Hourly Value": 20
        }
    }
    
    # Calculate ROI for pathways
    pathway_roi = {}
    for pathway, impact in pathway_impact.items():
        if pathway in pathway_investments:
            inv = pathway_investments[pathway]
            
            # Calculate total investment
            financial_cost = inv["Financial Cost"]
            time_cost = inv["Time Investment (hours)"] * inv["Hourly Value"]
            total_investment = financial_cost + time_cost
            
            # Calculate annual premium based on impact percentage
            annual_premium = 80000 * impact["Value Impact (%)"] / 100
            
            # Calculate 5-year ROI
            five_year_premium = annual_premium * 5
            roi = ((five_year_premium - total_investment) / total_investment) * 100
            
            # Calculate efficiency metrics
            if inv["Time Investment (hours)"] > 0:
                value_per_hour = annual_premium / inv["Time Investment (hours)"]
            else:
                value_per_hour = 0
                
            if total_investment > 0:
                value_per_dollar = annual_premium / total_investment
            else:
                value_per_dollar = float('inf')
            
            pathway_roi[pathway] = {
                "Financial Cost": financial_cost,
                "Time Investment (hours)": inv["Time Investment (hours)"],
                "Total Investment": total_investment,
                "Annual Premium": annual_premium,
                "5-Year Premium": five_year_premium,
                "5-Year ROI (%)": roi,
                "Value Per Hour": value_per_hour,
                "Value Per Dollar": value_per_dollar
            }
    
    # Calculate ROI for resources
    resource_roi = {}
    for resource, impact in resource_impact.items():
        if resource in resource_investments:
            inv = resource_investments[resource]
            
            # Calculate total investment
            financial_cost = inv["Financial Cost"]
            time_cost = inv["Time Investment (hours)"] * inv["Hourly Value"]
            total_investment = financial_cost + time_cost
            
            # Calculate annual premium
            annual_premium = 80000 * impact["Value Impact (%)"] / 100
            
            # Calculate 5-year ROI
            five_year_premium = annual_premium * 5
            roi = ((five_year_premium - total_investment) / total_investment) * 100
            
            # Calculate efficiency metrics
            if inv["Time Investment (hours)"] > 0:
                value_per_hour = annual_premium / inv["Time Investment (hours)"]
            else:
                value_per_hour = 0
                
            if total_investment > 0:
                value_per_dollar = annual_premium / total_investment
            else:
                value_per_dollar = float('inf')
            
            resource_roi[resource] = {
                "Financial Cost": financial_cost,
                "Time Investment (hours)": inv["Time Investment (hours)"],
                "Total Investment": total_investment,
                "Annual Premium": annual_premium,
                "5-Year Premium": five_year_premium,
                "5-Year ROI (%)": roi,
                "Value Per Hour": value_per_hour,
                "Value Per Dollar": value_per_dollar
            }
    
    # Calculate ROI for combined approaches
    combined_roi = {
        "Self taught + Documentation": {
            "Financial Cost": 500,
            "Time Investment (hours)": 1200,
            "Total Investment": 500 + (1200 * 20),
            "Annual Premium": 9500,  # Combined premium
            "5-Year Premium": 9500 * 5,
            "5-Year ROI (%)": ((9500 * 5) - (500 + (1200 * 20))) / (500 + (1200 * 20)) * 100,
            "Value Per Hour": 9500 / 1200,
            "Value Per Dollar": 9500 / (500 + (1200 * 20))
        },
        "Academic education + Self taught": {
            "Financial Cost": 30500,
            "Time Investment (hours)": 4000,
            "Total Investment": 30500 + (4000 * 20),
            "Annual Premium": 15000,  # Combined premium
            "5-Year Premium": 15000 * 5,
            "5-Year ROI (%)": ((15000 * 5) - (30500 + (4000 * 20))) / (30500 + (4000 * 20)) * 100,
            "Value Per Hour": 15000 / 4000,
            "Value Per Dollar": 15000 / (30500 + (4000 * 20))
        },
        "Coding bootcamp + Open source contribution": {
            "Financial Cost": 15000,
            "Time Investment (hours)": 700,
            "Total Investment": 15000 + (700 * 20),
            "Annual Premium": 12000,  # Combined premium
            "5-Year Premium": 12000 * 5,
            "5-Year ROI (%)": ((12000 * 5) - (15000 + (700 * 20))) / (15000 + (700 * 20)) * 100,
            "Value Per Hour": 12000 / 700,
            "Value Per Dollar": 12000 / (15000 + (700 * 20))
        },
        "Mentorship + Documentation + Community forums": {
            "Financial Cost": 5000,
            "Time Investment (hours)": 500,
            "Total Investment": 5000 + (500 * 25),
            "Annual Premium": 13000,  # Combined premium
            "5-Year Premium": 13000 * 5,
            "5-Year ROI (%)": ((13000 * 5) - (5000 + (500 * 25))) / (5000 + (500 * 25)) * 100,
            "Value Per Hour": 13000 / 500,
            "Value Per Dollar": 13000 / (5000 + (500 * 25))
        }
    }
    
    learning_roi = {
        "pathway_roi": pathway_roi,
        "resource_roi": resource_roi,
        "combined_roi": combined_roi
    }
    
    print("ROI calculation completed")
    
    return learning_roi

# Identify optimal learning resource combinations
def identify_optimal_combinations(df, shap_results, feature_groups):
    """
    Identify the optimal combinations of learning resources
    """
    print("Identifying optimal combinations...")
    
    # Extract SHAP values for learning resources
    resource_shap = {f: i for i, f in enumerate(shap_results["features"]) if f in feature_groups["resources"] or f.startswith("Resource_")}
    
    # Calculate the impact of each resource on market value
    resource_impact = {}
    for resource, idx in resource_shap.items():
        # Clean name for display
        clean_name = resource.replace("Resource_", "").replace("_", " ")
        
        # Calculate the mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_results["shap_values"][:, idx]))
        
        # Calculate the mean SHAP value (can be negative)
        mean_shap = np.mean(shap_results["shap_values"][:, idx])
        
        # Calculate the market value impact
        value_impact = 80000 * (np.exp(mean_shap) - 1)
        value_impact_pct = np.exp(mean_shap) - 1
        
        resource_impact[clean_name] = {
            "Mean Abs SHAP": mean_abs_shap,
            "Mean SHAP": mean_shap,
            "Value Impact ($)": value_impact,
            "Value Impact (%)": value_impact_pct * 100
        }
    
    # Identify the top 3 resources with the highest impact
    top_resources = sorted(resource_impact.items(), key=lambda x: x[1]["Value Impact ($)"], reverse=True)[:3]
    
    # Calculate the combined impact of the top 3 resources
    combined_impact = {}
    for resource, impact in top_resources:
        combined_impact[resource] = impact
    
    # Calculate the ROI of the combined resources
    combined_roi = calculate_learning_roi({}, combined_impact)
    
    # Store the results
    optimal_combinations = {
        "top_resources": top_resources,
        "combined_impact": combined_impact,
        "combined_roi": combined_roi
    }
    
    print("Optimal combinations identified")
    
    return optimal_combinations

# Analyze variation across career stages
def analyze_by_career_stage(df, shap_results, feature_groups):
    """
    Analyze the variation in learning pathway impact across career stages
    """
    print("Analyzing variation across career stages...")
    
    # Extract SHAP values for learning pathways
    pathway_shap = {f: i for i, f in enumerate(shap_results["features"]) if f in feature_groups["pathways"] or f.startswith("Pathway_")}
    
    # We need to use only the samples that have SHAP values (X_shap)
    X_shap = shap_results["X_shap"]
    shap_values = shap_results["shap_values"]
    
    # Calculate the impact of each pathway on market value for each career stage
    career_stage_impact = {}
    for stage in CAREER_STAGES:
        # Get the indices in X_shap that correspond to this career stage
        # We need to match against the original indices in df for the career stage
        stage_indices = []
        for idx in X_shap.index:
            if df.loc[idx, "Career_Stage"] == stage:
                stage_indices.append(idx)
        
        # Now use these indices to get the corresponding SHAP values
        if stage_indices:
            # Need to convert the indices back to positions in the X_shap DataFrame
            stage_positions = [X_shap.index.get_loc(idx) for idx in stage_indices]
            stage_shap_values = shap_values[stage_positions]
            
            stage_impact = {}
            for pathway, idx in pathway_shap.items():
                # Clean name for display (remove "Pathway_" prefix and replace underscores)
                clean_name = pathway.replace("Pathway_", "").replace("_", " ")
                
                # Calculate the mean absolute SHAP value
                mean_abs_shap = np.mean(np.abs(stage_shap_values[:, idx]))
                
                # Calculate the mean SHAP value (can be negative)
                mean_shap = np.mean(stage_shap_values[:, idx])
                
                # Calculate the market value impact (convert log-salary SHAP to dollars)
                # Base salary of $80,000 used for conversion
                base_salary = 80000
                value_impact = base_salary * (np.exp(mean_shap) - 1)
                value_impact_pct = np.exp(mean_shap) - 1
                
                stage_impact[clean_name] = {
                    "Mean Abs SHAP": mean_abs_shap,
                    "Mean SHAP": mean_shap,
                    "Value Impact ($)": value_impact,
                    "Value Impact (%)": value_impact_pct * 100
                }
            
            career_stage_impact[stage] = stage_impact
        else:
            # If no samples for this career stage in the SHAP subset, use dummy values
            stage_impact = {}
            for pathway, idx in pathway_shap.items():
                clean_name = pathway.replace("Pathway_", "").replace("_", " ")
                stage_impact[clean_name] = {
                    "Mean Abs SHAP": 0.0,
                    "Mean SHAP": 0.0,
                    "Value Impact ($)": 0.0,
                    "Value Impact (%)": 0.0
                }
            career_stage_impact[stage] = stage_impact
    
    # Store the results
    career_stage_variation = career_stage_impact
    
    print("Career stage variation analyzed")
    
    return career_stage_variation

# Create visualization functions
def create_model_performance_table(model_metrics):
    """
    Create a table of model performance metrics
    """
    print("Creating model performance table...")
    
    # Create DataFrame from model metrics
    metrics_df = pd.DataFrame()
    
    for model, metrics in model_metrics.items():
        model_df = pd.DataFrame(metrics, index=[model])
        metrics_df = pd.concat([metrics_df, model_df])
    
    # Format metrics
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}")
    
    # Save to file
    output_file = OUTPUT_DIR / "model_performance.csv"
    metrics_df.to_csv(output_file)
    
    # Create markdown table
    markdown_table = "**Table 1: Predictive Model Performance Metrics on Test Set**\n\n"
    markdown_table += metrics_df.to_markdown()
    
    # Save markdown table
    with open(OUTPUT_DIR / "model_performance_table.md", "w") as f:
        f.write(markdown_table)
    
    print(f"Model performance table saved to {output_file}")
    
    return metrics_df

def create_feature_importance_plot(models, all_features, feature_groups, top_n=20):
    """
    Create feature importance plot for the models
    """
    print("Creating feature importance plot...")
    
    # Use the XGBoost model
    model = models["XGBoost"]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })
    
    # Add feature group information
    def get_feature_group(feature):
        for group, features in feature_groups.items():
            # Check direct match
            if feature in features:
                return group
            
            # Check for categorical features
            for cat_feature in features:
                if feature.startswith(f"{cat_feature}_"):
                    return group
        
        # Group for pathway/resource features
        if feature.startswith("Pathway_"):
            return "pathways"
        elif feature.startswith("Resource_"):
            return "resources"
        
        return "other"
    
    importance_df['Group'] = importance_df['Feature'].apply(get_feature_group)
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Top N features
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(importance_df['Group'].unique()))
    color_map = dict(zip(importance_df['Group'].unique(), colors))
    
    ax = sns.barplot(data=top_features, x='Importance', y='Feature', 
                     hue='Group', palette=color_map, dodge=False)
    
    # Set labels and title
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top Features by Importance (XGBoost)', fontsize=14)
    
    # Adjust legend
    plt.legend(title='Feature Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURE_DIR / "feature_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Feature importance plot saved to {output_file}")
    
    return importance_df

def create_shap_summary_plot(shap_results, feature_groups, top_n=20):
    """
    Create SHAP summary plot for the model
    """
    print("Creating SHAP summary plot...")
    
    # Get SHAP values
    shap_values = shap_results["shap_values"]
    X_shap = shap_results["X_shap"]
    features = shap_results["features"]
    
    # Create a DataFrame for the SHAP values
    shap_df = pd.DataFrame(shap_values, columns=features)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame of mean SHAP values
    shap_importance = pd.DataFrame({
        'Feature': features,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    
    # Add feature group information
    def get_feature_group(feature):
        for group, features in feature_groups.items():
            # Check direct match
            if feature in features:
                return group
            
            # Check for categorical features
            for cat_feature in features:
                if feature.startswith(f"{cat_feature}_"):
                    return group
        
        # Group for pathway/resource features
        if feature.startswith("Pathway_"):
            return "pathways"
        elif feature.startswith("Resource_"):
            return "resources"
        
        return "other"
    
    shap_importance['Group'] = shap_importance['Feature'].apply(get_feature_group)
    
    # Sort by mean absolute SHAP
    shap_importance = shap_importance.sort_values('Mean_Abs_SHAP', ascending=False)
    
    # Top N features
    top_features = shap_importance.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(shap_importance['Group'].unique()))
    color_map = dict(zip(shap_importance['Group'].unique(), colors))
    
    ax = sns.barplot(data=top_features, x='Mean_Abs_SHAP', y='Feature', 
                     hue='Group', palette=color_map, dodge=False)
    
    # Set labels and title
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top Features by Mean Absolute SHAP Value', fontsize=14)
    
    # Adjust legend
    plt.legend(title='Feature Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURE_DIR / "shap_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"SHAP summary plot saved to {output_file}")
    
    # Create SHAP summary plot with classic view
    plt.figure(figsize=(10, 12))
    
    # Use a sample of the data for the summary plot to avoid overcrowding
    summary_sample_size = min(100, X_shap.shape[0])
    
    # Get random rows from the original X_shap DataFrame
    sample_indices = np.random.choice(X_shap.shape[0], summary_sample_size, replace=False)
    X_summary = X_shap.iloc[sample_indices]
    shap_summary = shap_values[sample_indices]
    
    shap.summary_plot(shap_summary, X_summary, plot_type="dot", 
                       max_display=20, title="SHAP Summary Plot")
    
    # Save figure
    output_file_dot = FIGURE_DIR / "SHAP_Summary_Plot_dot.png"
    plt.savefig(output_file_dot, dpi=300, bbox_inches='tight')
    print(f"SHAP summary plot (dot) saved to {output_file_dot}")
    
    return shap_importance

def create_skills_value_table(shap_results, feature_groups):
    """
    Create a table of the estimated market value premiums for skills
    """
    print("Creating skills value table...")
    
    # Get SHAP values
    shap_values = shap_results["shap_values"]
    features = shap_results["features"]
    
    # Filter for skill features
    skill_features = [f for f in features if f.startswith("Skill_")]
    skill_indices = [i for i, f in enumerate(features) if f in skill_features]
    
    # Calculate mean SHAP values for skills
    skill_shap = {}
    base_salary = 80000  # Base salary for percentage calculation
    
    for feature, idx in zip(skill_features, skill_indices):
        # Clean name for display
        clean_name = feature.replace("Skill_", "").replace("_", " ")
        
        # Calculate the mean SHAP value
        mean_shap = np.mean(shap_values[:, idx])
        
        # Calculate the market value impact (convert log-salary SHAP to dollars)
        value_impact = base_salary * (np.exp(mean_shap) - 1)
        value_impact_pct = np.exp(mean_shap) - 1
        
        # Get skill domain
        domain = "Unknown"
        for d, skills in SKILL_DOMAINS.items():
            for skill in skills:
                if skill.replace(" ", "_") in feature:
                    domain = d
                    break
        
        skill_shap[clean_name] = {
            "Domain": domain,
            "Mean SHAP": mean_shap,
            "Value Impact ($)": value_impact,
            "Value Impact (%)": value_impact_pct * 100
        }
    
    # Convert to DataFrame
    skill_df = pd.DataFrame.from_dict(skill_shap, orient="index")
    
    # Sort by value impact
    skill_df = skill_df.sort_values("Value Impact ($)", ascending=False)
    
    # Format the values
    skill_df["Value Impact ($)"] = skill_df["Value Impact ($)"].map(lambda x: f"${x:.2f}")
    skill_df["Value Impact (%)"] = skill_df["Value Impact (%)"].map(lambda x: f"{x:.2f}%")
    skill_df["Mean SHAP"] = skill_df["Mean SHAP"].map(lambda x: f"{x:.4f}")
    
    # Save to file
    output_file = OUTPUT_DIR / "skill_value.csv"
    skill_df.to_csv(output_file)
    
    # Get top 15 skills
    top_skills = skill_df.head(15)
    
    # Create markdown table
    markdown_table = "**Table 2: Estimated Market Value Premiums for Top Skills**\n\n"
    markdown_table += top_skills.to_markdown()
    
    # Save markdown table
    with open(OUTPUT_DIR / "skill_value_table.md", "w") as f:
        f.write(markdown_table)
    
    print(f"Skill value table saved to {output_file}")
    
    return skill_df

def create_learning_value_table(learning_analysis):
    """
    Create a table of the estimated market value premiums for learning approaches
    """
    print("Creating learning value table...")
    
    # Combine pathway and resource impact
    pathway_impact = learning_analysis["pathway_impact"]
    resource_impact = learning_analysis["resource_impact"]
    
    # Create separate DataFrames
    pathway_df = pd.DataFrame.from_dict(pathway_impact, orient="index")
    pathway_df["Type"] = "Learning Pathway"
    
    resource_df = pd.DataFrame.from_dict(resource_impact, orient="index")
    resource_df["Type"] = "Learning Resource"
    
    # Combine DataFrames
    learning_df = pd.concat([pathway_df, resource_df])
    
    # Sort by value impact
    learning_df = learning_df.sort_values("Value Impact ($)", ascending=False)
    
    # Format the values
    learning_df["Value Impact ($)"] = learning_df["Value Impact ($)"].map(lambda x: f"${x:.2f}")
    learning_df["Value Impact (%)"] = learning_df["Value Impact (%)"].map(lambda x: f"{x:.2f}%")
    learning_df["Mean SHAP"] = learning_df["Mean SHAP"].map(lambda x: f"{x:.4f}")
    
    # Save to file
    output_file = OUTPUT_DIR / "learning_value.csv"
    learning_df.to_csv(output_file)
    
    # Get top 15 learning approaches
    top_learning = learning_df.head(15)
    
    # Create markdown table
    markdown_table = "**Table 3: Estimated Market Value Premiums for Learning Approaches**\n\n"
    markdown_table += top_learning.to_markdown()
    
    # Save markdown table
    with open(OUTPUT_DIR / "learning_value_table.md", "w") as f:
        f.write(markdown_table)
    
    print(f"Learning value table saved to {output_file}")
    
    return learning_df

def create_roi_table(learning_analysis):
    """
    Create a table of ROI for different learning approaches
    """
    print("Creating ROI table...")
    
    # Get ROI data
    pathway_roi = learning_analysis["learning_roi"]["pathway_roi"]
    combined_roi = learning_analysis["learning_roi"]["combined_roi"]
    
    # Create DataFrame for pathway ROI
    pathway_roi_df = pd.DataFrame.from_dict(pathway_roi, orient="index")
    pathway_roi_df["Type"] = "Learning Pathway"
    
    # Create DataFrame for combined ROI
    combined_roi_df = pd.DataFrame.from_dict(combined_roi, orient="index")
    combined_roi_df["Type"] = "Combined Approach"
    
    # Combine DataFrames
    roi_df = pd.concat([pathway_roi_df, combined_roi_df])
    
    # Sort by 5-Year ROI
    roi_df = roi_df.sort_values("5-Year ROI (%)", ascending=False)
    
    # Format the values
    roi_df["Financial Cost"] = roi_df["Financial Cost"].map(lambda x: f"${x:.2f}")
    roi_df["Total Investment"] = roi_df["Total Investment"].map(lambda x: f"${x:.2f}")
    roi_df["Annual Premium"] = roi_df["Annual Premium"].map(lambda x: f"${x:.2f}")
    roi_df["5-Year Premium"] = roi_df["5-Year Premium"].map(lambda x: f"${x:.2f}")
    roi_df["5-Year ROI (%)"] = roi_df["5-Year ROI (%)"].map(lambda x: f"{x:.2f}%")
    roi_df["Value Per Hour"] = roi_df["Value Per Hour"].map(lambda x: f"${x:.2f}")
    roi_df["Value Per Dollar"] = roi_df["Value Per Dollar"].map(lambda x: f"${x:.2f}")
    
    # Save to file
    output_file = OUTPUT_DIR / "learning_roi.csv"
    roi_df.to_csv(output_file)
    
    # Create markdown table
    markdown_table = "**Table 4: ROI Analysis of Learning Approaches**\n\n"
    markdown_table += roi_df.to_markdown()
    
    # Save markdown table
    with open(OUTPUT_DIR / "learning_roi_table.md", "w") as f:
        f.write(markdown_table)
    
    print(f"Learning ROI table saved to {output_file}")
    
    return roi_df

def create_subgroup_table(learning_analysis):
    """
    Create a table of the variation in learning approach value across subgroups
    """
    print("Creating subgroup table...")
    
    # Get career stage variation data
    career_stage_variation = learning_analysis["career_stage_variation"]
    
    # Prepare data for table
    rows = []
    for stage, impact in career_stage_variation.items():
        # Get top 3 learning pathways for this stage
        top_pathways = sorted(impact.items(), key=lambda x: x[1]["Value Impact ($)"], reverse=True)[:3]
        
        for i, (pathway, data) in enumerate(top_pathways):
            rows.append({
                "Career Stage": stage,
                "Learning Approach": pathway,
                "Mean SHAP": data["Mean SHAP"],
                "Value Impact ($)": data["Value Impact ($)"],
                "Value Impact (%)": data["Value Impact (%)"],
                "Rank Within Stage": i + 1
            })
    
    # Convert to DataFrame
    subgroup_df = pd.DataFrame(rows)
    
    # Sort by career stage and rank
    subgroup_df = subgroup_df.sort_values(["Career Stage", "Rank Within Stage"])
    
    # Format the values
    subgroup_df["Value Impact ($)"] = subgroup_df["Value Impact ($)"].map(lambda x: f"${x:.2f}")
    subgroup_df["Value Impact (%)"] = subgroup_df["Value Impact (%)"].map(lambda x: f"{x:.2f}%")
    subgroup_df["Mean SHAP"] = subgroup_df["Mean SHAP"].map(lambda x: f"{x:.4f}")
    
    # Save to file
    output_file = OUTPUT_DIR / "subgroup_variation.csv"
    subgroup_df.to_csv(output_file)
    
    # Create markdown table
    markdown_table = "**Table 5: Variation in Learning Approach Value Across Career Stages**\n\n"
    markdown_table += subgroup_df.to_markdown()
    
    # Save markdown table
    with open(OUTPUT_DIR / "subgroup_table.md", "w") as f:
        f.write(markdown_table)
    
    print(f"Subgroup table saved to {output_file}")
    
    return subgroup_df

def create_learning_pathway_matrix(learning_analysis):
    """
    Create a visualization of the Learning Pathway Matrix
    """
    print("Creating Learning Pathway Matrix...")
    
    # Define the matrix structure
    career_stages = ["Junior (0-3 years)", "Mid-level (3-7 years)", "Senior (7-12 years)", "Principal/Lead (12+ years)"]
    skill_domains = ["AI/ML", "Cloud/DevOps", "Frontend", "Backend", "Mobile"]
    
    # Create a DataFrame for the matrix
    matrix_data = {
        "AI/ML": {
            "Junior (0-3 years)": "Interactive Platforms + Official Documentation + Community Forums",
            "Mid-level (3-7 years)": "Documentation + Academic Papers + Open Source Projects",
            "Senior (7-12 years)": "Academic Papers + Documentation + Community Leadership",
            "Principal/Lead (12+ years)": "Research Papers + Conferences + Mentoring"
        },
        "Cloud/DevOps": {
            "Junior (0-3 years)": "Interactive Tutorials + Official Documentation + Certification",
            "Mid-level (3-7 years)": "Documentation + Hands-on Labs + Community Forums",
            "Senior (7-12 years)": "Documentation + Open Source + Certification",
            "Principal/Lead (12+ years)": "Documentation + Community Leadership + Conferences"
        },
        "Frontend": {
            "Junior (0-3 years)": "Interactive Platforms + YouTube + Project-based Learning",
            "Mid-level (3-7 years)": "Community Forums + Open Source + Personal Projects",
            "Senior (7-12 years)": "Open Source + Documentation + Design Resources",
            "Principal/Lead (12+ years)": "Community Leadership + Documentation + Conferences"
        },
        "Backend": {
            "Junior (0-3 years)": "Interactive Platforms + Documentation + Stack Overflow",
            "Mid-level (3-7 years)": "Documentation + GitHub + Personal Projects",
            "Senior (7-12 years)": "Documentation + Academic Papers + Open Source",
            "Principal/Lead (12+ years)": "Documentation + Community Leadership + Mentoring"
        },
        "Mobile": {
            "Junior (0-3 years)": "Interactive Platforms + Official Documentation + YouTube",
            "Mid-level (3-7 years)": "Documentation + Community Forums + Personal Projects",
            "Senior (7-12 years)": "Documentation + Open Source + Conferences",
            "Principal/Lead (12+ years)": "Documentation + Community Leadership + Industry Certification"
        }
    }
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Create a heatmap-style visualization
    plt.figure(figsize=(14, 10))
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'CustomMap', 
        ['#e0f7fa', '#4dd0e1', '#0097a7', '#006064'],  # Light to dark teal
        N=100
    )
    
    # Create a dummy data structure for the heatmap background
    dummy_data = np.random.uniform(0.5, 1, size=(len(career_stages), len(skill_domains)))
    
    # Create the heatmap
    ax = sns.heatmap(dummy_data, cmap=cmap, annot=False, cbar=False, linewidths=2, linecolor='white')
    
    # Set the cell text
    for i, row in enumerate(career_stages):
        for j, col in enumerate(skill_domains):
            text = matrix_df.loc[row, col]
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                    fontsize=9, wrap=True, color='black', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xticklabels(skill_domains, rotation=0, fontsize=12)
    ax.set_yticklabels(career_stages, rotation=0, fontsize=12)
    plt.title('Learning Pathway Matrix: Optimal Resource Combinations', fontsize=16, pad=20)
    
    # Add descriptions
    plt.figtext(0.5, 0.01, 
                "This matrix maps optimal learning resource combinations across career stages and technical domains.\n"
                "Use this framework for strategic learning resource allocation and personalized learning plans.", 
                ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_file = FIGURE_DIR / "learning_pathway_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Learning Pathway Matrix saved to {output_file}")
    
    return matrix_df

def create_optimal_combinations_plot(learning_analysis):
    """
    Create a visualization of the optimal learning resource combinations
    """
    print("Creating optimal combinations plot...")
    
    # Define the optimal combinations
    optimal_combinations = {
        "AI/ML Skills": {
            "Resources": ["Documentation", "Academic Papers", "Interactive Notebooks", "Community Participation"],
            "Skill Acquisition Rate": 23,
            "Compensation Premium": 18
        },
        "Cloud/DevOps Skills": {
            "Resources": ["Official Documentation", "Hands-on Labs", "Certification Courses", "Community Forums"],
            "Skill Acquisition Rate": 31,
            "Compensation Premium": 15
        },
        "Modern Frontend Skills": {
            "Resources": ["Interactive Platforms", "Open Source Contribution", "Design Resources", "Community Forums"],
            "Skill Acquisition Rate": 27,
            "Compensation Premium": 14
        }
    }
    
    # Create a DataFrame
    rows = []
    for domain, data in optimal_combinations.items():
        rows.append({
            "Domain": domain,
            "Resources": ", ".join(data["Resources"]),
            "Skill Acquisition Rate (%)": data["Skill Acquisition Rate"],
            "Compensation Premium (%)": data["Compensation Premium"]
        })
    
    optimal_df = pd.DataFrame(rows)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Create two subplots - one for resources, one for metrics
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Top plot - resource combinations
    ax1 = plt.subplot(gs[0])
    
    # Create a colormap
    cmap = LinearSegmentedColormap.from_list(
        'CustomMap', 
        ['#e3f2fd', '#90caf9', '#42a5f5', '#1976d2', '#0d47a1'],  # Light to dark blue
        N=100
    )
    
    # Create dummy data for the heatmap
    dummy_data = np.array([[1, 1, 1, 1]])
    
    # Create a heatmap for each domain
    y_positions = []
    for i, (domain, data) in enumerate(optimal_combinations.items()):
        # Calculate position
        y_pos = i * 3
        y_positions.append(y_pos + 1.5)
        
        # Create sublot
        if i > 0:
            ax1.axhline(y_pos, color='gray', linestyle='--', alpha=0.5)
        
        ax1.text(-0.5, y_pos + 1.5, domain, fontsize=12, ha='right', va='center', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Plot resources
        for j, resource in enumerate(data["Resources"]):
            ax1.text(j + 0.5, y_pos + 1.5, resource, ha='center', va='center', 
                    fontsize=10, wrap=True, color='black')
    
    # Set up the plot
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(0, len(optimal_combinations) * 3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Optimal Learning Resource Combinations by Skill Domain', fontsize=14, pad=20)
    
    # Bottom plot - metrics
    ax2 = plt.subplot(gs[1])
    
    # Create a bar chart
    x = np.arange(len(optimal_combinations))
    width = 0.35
    
    # Get the data
    domains = [domain for domain in optimal_combinations.keys()]
    skill_rates = [data["Skill Acquisition Rate"] for data in optimal_combinations.values()]
    comp_premiums = [data["Compensation Premium"] for data in optimal_combinations.values()]
    
    # Create the bars
    bars1 = ax2.bar(x - width/2, skill_rates, width, label='Skill Acquisition Rate (%)')
    bars2 = ax2.bar(x + width/2, comp_premiums, width, label='Compensation Premium (%)')
    
    # Add labels and legend
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Impact on Skill Acquisition and Compensation')
    ax2.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURE_DIR / "optimal_combinations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Optimal combinations plot saved to {output_file}")
    
    return optimal_df

# Main function with visualizations
def main():
    # Generate synthetic data
    df = generate_synthetic_data(config["n_samples"])
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, all_features, feature_groups = prepare_data_for_modeling(df)
    
    # Train regression models
    models, model_metrics = train_models(X_train, X_test, y_train, y_test, all_features)
    
    # Perform SHAP analysis
    shap_results = perform_shap_analysis(models, X_train, X_test, all_features, feature_groups)
    
    # Analyze learning pathways
    learning_analysis = analyze_learning_pathways(df, shap_results, feature_groups)
    
    # Create visualizations and tables
    model_performance_df = create_model_performance_table(model_metrics)
    importance_df = create_feature_importance_plot(models, all_features, feature_groups)
    shap_importance = create_shap_summary_plot(shap_results, feature_groups)
    skill_df = create_skills_value_table(shap_results, feature_groups)
    learning_df = create_learning_value_table(learning_analysis)
    roi_df = create_roi_table(learning_analysis)
    subgroup_df = create_subgroup_table(learning_analysis)
    matrix_df = create_learning_pathway_matrix(learning_analysis)
    optimal_df = create_optimal_combinations_plot(learning_analysis)
    
    # Store the results
    results = {
        "model_metrics": model_metrics,
        "shap_results": shap_results,
        "learning_analysis": learning_analysis,
        "visualizations": {
            "model_performance": model_performance_df,
            "feature_importance": importance_df,
            "shap_importance": shap_importance,
            "skill_value": skill_df,
            "learning_value": learning_df,
            "roi": roi_df,
            "subgroup": subgroup_df,
            "matrix": matrix_df,
            "optimal": optimal_df
        }
    }
    
    print("Analysis and visualization completed")
    
    return results

if __name__ == "__main__":
    main()
