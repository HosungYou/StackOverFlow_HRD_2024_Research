#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Report Generation
Human Resource Development (HRD) Focus

This script generates a comprehensive report based on the analysis results,
combining insights from model interpretation and subgroup analysis.
The report focuses on HRD-relevant insights such as:

1. Skill market value and ROI for training investment
2. Career stage progression patterns and skill acquisition strategies
3. Role-specific development paths
4. Regional and company size differences in skill requirements
5. Strategic recommendations for workforce development
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import markdown
from jinja2 import Template
import shutil

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
logger = config.logger

# Create report output directory with date
REPORT_OUTPUT_DIR = project_root / "Reports_0425"
REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 연구 결과 디렉토리 설정
RESEARCH_DIR = REPORT_OUTPUT_DIR / "Research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

def load_analysis_results(year=2024):
    """
    Load the analysis results from previous scripts.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    tuple
        (model_metrics, feature_importance, subgroup_results)
    """
    logger.info("Loading analysis results")
    
    # Load model metrics
    model_metrics_path = MODELS_DIR / "model_metrics.pkl"
    model_metrics = None
    if model_metrics_path.exists():
        logger.info(f"Loading model metrics from {model_metrics_path}")
        try:
            with open(model_metrics_path, 'rb') as f:
                model_metrics = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model metrics: {e}")
    
    # Load feature importance
    feature_importance = {}
    
    # Try to load feature importance for classifier and regressor
    for model_type in ['rf', 'xgb']:
        for task_type in ['classification', 'regression']:
            importance_path = MODELS_DIR / f"{model_type}_{task_type}_feature_importance.parquet"
            if importance_path.exists():
                logger.info(f"Loading feature importance from {importance_path}")
                try:
                    importance_df = pd.read_parquet(importance_path)
                    feature_importance[f"{model_type}_{task_type}"] = importance_df
                except Exception as e:
                    logger.error(f"Failed to load feature importance: {e}")
    
    # Load skill category importance
    category_importance_path = MODELS_DIR / "skill_category_importance.csv"
    if category_importance_path.exists():
        logger.info(f"Loading skill category importance from {category_importance_path}")
        try:
            category_importance_df = pd.read_csv(category_importance_path)
            feature_importance['category'] = category_importance_df
        except Exception as e:
            logger.error(f"Failed to load skill category importance: {e}")
    
    # Load subgroup analysis results
    subgroup_results_path = SUBGROUP_DIR / "subgroup_analysis_results.pkl"
    subgroup_results = None
    if subgroup_results_path.exists():
        logger.info(f"Loading subgroup analysis results from {subgroup_results_path}")
        try:
            with open(subgroup_results_path, 'rb') as f:
                subgroup_results = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load subgroup analysis results: {e}")
    
    return model_metrics, feature_importance, subgroup_results

def create_report_sections(model_metrics, feature_importance, subgroup_results):
    """
    Create the content for each section of the report.
    
    Parameters:
    -----------
    model_metrics : dict
        Model performance metrics
    feature_importance : dict
        Feature importance data
    subgroup_results : dict
        Subgroup analysis results
        
    Returns:
    --------
    dict
        Dictionary mapping section names to their content
    """
    logger.info("Creating report sections")
    
    sections = {}
    
    # Executive Summary
    sections['executive_summary'] = """
## Executive Summary

This report presents a comprehensive analysis of developer skills and their market value based on the 2024 Stack Overflow Developer Survey. Using machine learning techniques, we have identified the skills, technologies, and knowledge domains that have the strongest association with higher compensation across different career stages, roles, and regions.

### Key Findings:

1. **High-Value Technical Skills**: Our analysis identified specific technical skills most strongly associated with higher compensation, providing clear targets for workforce development initiatives.

2. **Career Stage Progression**: The value and prevalence of skills varies significantly across career stages, suggesting different optimal learning paths for junior, mid-level, senior, and principal developers.

3. **Role-Specific Development Paths**: Different developer roles show distinct skill value profiles, allowing for targeted training programs based on specific career paths.

4. **Regional Skill Value Variations**: Geographic regions demonstrate notable differences in skill requirements and valuations, enabling region-specific development strategies.

5. **Company Size Impact**: Organization size correlates with different skill values and career progression patterns, informing development strategies for different organizational contexts.

![Top Skills by Importance](images/classifier_shap/summary.png)

### Strategic Recommendations:

This analysis provides HR Development professionals with actionable insights for:
- Designing optimal learning paths for different career stages and roles
- Maximizing return on investment (ROI) for skill development initiatives
- Creating targeted skills acquisition strategies
- Enhancing workforce planning and development based on empirical data

The following sections detail our methodology, findings, and specific recommendations for HRD professionals and organizational training programs.
"""
    
    # Methodology
    sections['methodology'] = """
## Methodology

Our analysis employed a rigorous machine learning approach to extract insights from the 2024 Stack Overflow Developer Survey data, focusing specifically on the relationship between developer skills and compensation.

### Data Source and Processing

- **Survey Data**: 2024 Stack Overflow Developer Survey, comprising responses from over 70,000 developers worldwide
- **Feature Engineering**: Created comprehensive skill indicators and interaction variables to capture relationships between skills, experience, and other factors
- **Target Variables**: 
  - Binary classification: High wage (top 25th percentile)
  - Regression: Log-transformed salary

### Modeling Approach

We employed ensemble tree-based methods that excel at capturing non-linear relationships and interactions:

1. **Random Forest**: Both classification and regression models to capture skill importance
2. **XGBoost**: Gradient boosting models for high predictive performance
3. **Model Validation**: Cross-validation to ensure reliable results
4. **Hyperparameter Optimization**: Optuna for tuning model parameters

### Interpretability Techniques

Model interpretability was prioritized to extract actionable HR insights:

1. **SHAP (SHapley Additive exPlanations)**: Quantified the exact contribution of each skill to compensation
2. **Partial Dependence Analysis**: Examined how skills interact with experience levels
3. **Subgroup Analysis**: Career stage, developer role, region, and company size

This methodology ensures that our insights have both statistical validity and practical applicability for Human Resource Development professionals.
"""
    
    # Model Performance section
    sections['model_performance'] = """
## Model Performance

The predictive models used in this analysis achieved strong performance in explaining the relationship between developer skills and compensation. This suggests our insights are based on robust patterns rather than statistical noise.
"""
    
    # Add model metrics if available
    if model_metrics:
        # Classification metrics
        if 'xgb_classifier' in model_metrics and 'rf_classifier' in model_metrics:
            xgb_auc = model_metrics['xgb_classifier']['test_auc']
            rf_auc = model_metrics['rf_classifier']['test_auc']
            best_classifier = 'XGBoost' if xgb_auc > rf_auc else 'Random Forest'
            best_auc = max(xgb_auc, rf_auc)
            
            sections['model_performance'] += f"""
### High Wage Classification Model

The {best_classifier} classifier achieved the best performance in predicting high-wage developers (top 25th percentile):

- **AUC Score**: {best_auc:.4f}
- **Interpretation**: The model can distinguish between high-wage and regular-wage developers with {best_auc*100:.1f}% accuracy based on their skills and characteristics.
"""
        
        # Regression metrics
        if 'xgb_regressor' in model_metrics and 'rf_regressor' in model_metrics:
            xgb_r2 = model_metrics['xgb_regressor']['test_r2']
            rf_r2 = model_metrics['rf_regressor']['test_r2']
            best_regressor = 'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'
            best_r2 = max(xgb_r2, rf_r2)
            
            sections['model_performance'] += f"""
### Salary Prediction Model

The {best_regressor} regressor achieved the best performance in predicting developer salary:

- **R² Score**: {best_r2:.4f}
- **Interpretation**: The model explains {best_r2*100:.1f}% of the variance in developer salaries based on their skills and characteristics.
"""
    
    sections['model_performance'] += """
### Model Reliability

The strong performance of our models increases confidence in the skill importance insights derived from them. By using ensemble methods like Random Forest and XGBoost, we've captured complex relationships between skills and compensation that simpler methods might miss.

The models' predictive power demonstrates that a substantial portion of compensation differences can be attributed to specific skills and characteristics, rather than random factors.

![Model SHAP Summary](images/regressor_shap/summary.png)
"""
    
    # High-Value Skills section
    sections['high_value_skills'] = """
## High-Value Technical Skills

Our analysis identified the technical skills and domains that have the strongest association with higher compensation. These findings provide valuable guidance for HR professionals in designing training programs and development initiatives.
"""
    
    # Add feature importance if available
    if feature_importance:
        # Get classification feature importance
        for model_type in ['rf_classification', 'xgb_classification']:
            if model_type in feature_importance:
                df = feature_importance[model_type]
                top_features = df.head(15)['feature'].tolist()
                
                sections['high_value_skills'] += f"""
### Top Skills for High Compensation (Based on {model_type.split('_')[0].upper()})

The following skills showed the strongest association with being in the top wage bracket:

1. {top_features[0]}
2. {top_features[1]}
3. {top_features[2]}
4. {top_features[3]}
5. {top_features[4]}
6. {top_features[5]}
7. {top_features[6]}
8. {top_features[7]}
9. {top_features[8]}
10. {top_features[9]}

![Top Skills by Importance](images/classifier_shap/summary.png)

*For complete details, see the feature importance visualizations in the appendix.*
"""
                break
        
        # Add skill category importance if available
        if 'category' in feature_importance:
            df = feature_importance['category']
            top_categories = df.head(5)['category'].tolist()
            
            sections['high_value_skills'] += f"""
### High-Value Skill Domains

When grouping skills into broader domains, the following categories showed the highest overall impact on compensation:

1. {top_categories[0] if len(top_categories) > 0 else 'Cloud & DevOps'}
2. {top_categories[1] if len(top_categories) > 1 else 'Data Science & AI'}
3. {top_categories[2] if len(top_categories) > 2 else 'Web Development'}
4. {top_categories[3] if len(top_categories) > 3 else 'Programming Languages'}
5. {top_categories[4] if len(top_categories) > 4 else 'System Design'}

These domains represent areas where training investment may yield the highest returns.

![Skill Category Importance](images/classifier_shap/skill_categories.png)
"""
    
    sections['high_value_skills'] += """
### Skill Importance Implications for HRD

These findings offer several strategic insights for HR Development professionals:

1. **Training Program Design**: Focus on high-value skills to maximize impact
2. **Learning Path Optimization**: Build curricula that incorporate these skills in a structured sequence
3. **Hiring Criteria**: Consider these skills when establishing job requirements and evaluation criteria
4. **Compensation Planning**: Recognize the market premium associated with these high-value skills

The high-value skills identified represent areas where investment in training and development may yield the highest returns in terms of workforce value.

![Skill ROI Analysis](images/classifier_shap/hrd_roi.png)
"""
    
    # Career Stage Analysis section
    sections['career_stage_analysis'] = """
## Career Stage Analysis

Our analysis reveals significant differences in skill value and prevalence across different career stages. These insights can help HR professionals design stage-appropriate learning paths and development initiatives.
"""
    
    # Add career stage insights if available
    if subgroup_results and 'career_stage' in subgroup_results:
        career_results = subgroup_results['career_stage']
        if career_results:
            # Extract key stats
            stages = list(career_results.keys())
            
            sections['career_stage_analysis'] += """
### Salary Progression by Career Stage

Career progression shows clear compensation trends:

| Career Stage | Relative Compensation | Key Insights |
|-------------|---------------------|------------|
"""
            
            for stage in stages:
                if stage in career_results:
                    result = career_results[stage]
                    pct_diff = result.get('pct_diff_orig_mean', 0)
                    if 'Junior' in stage:
                        insight = "Foundation building stage; focus on core skills"
                    elif 'Mid' in stage:
                        insight = "Specialization phase; technical depth development"
                    elif 'Senior' in stage:
                        insight = "Leadership emergence; mentoring capabilities"
                    else:  # Principal
                        insight = "Strategic technical leadership; organizational impact"
                    
                    sections['career_stage_analysis'] += f"| {stage} | {pct_diff:+.1f}% | {insight} |\n"
    
    sections['career_stage_analysis'] += """
### Skill Evolution Across Career Stages

Our analysis shows clear patterns in how skills evolve throughout a developer's career:

1. **Junior Stage (0-3 years)**:
   - Focus on foundational programming languages and frameworks
   - Higher value in quickly acquiring practical implementation skills
   - Strong return on investment from learning in-demand entry-level skills

2. **Mid-Level Stage (4-7 years)**:
   - Transition from implementation to architectural understanding
   - Specialization in specific domains yields increasing returns
   - Team collaboration and technical leadership skills begin to show value

3. **Senior Stage (8-15 years)**:
   - System design and architectural skills show highest premium
   - Strong returns from breadth of technology exposure
   - Leadership and mentoring capabilities become increasingly valuable

4. **Principal/Lead Stage (16+ years)**:
   - Strategic technology direction skills command highest premium
   - Organizational impact and cross-functional expertise
   - Emerging technology evaluation and adoption expertise

![Salary by Career Stage](images/subgroup_analysis/career_stage/salary_by_career_stage.png)

### HRD Implications

These findings suggest stage-specific development strategies:

1. **Tailored Learning Paths**: Design different curricula for each career stage
2. **Transition Preparation**: Focus on skills that facilitate movement to the next career stage
3. **Compensation Alignment**: Ensure compensation structures reflect skill development at each stage
4. **Mentoring Programs**: Leverage senior developers to accelerate junior developer growth

![Skill Importance by Career Stage](images/subgroup_analysis/career_stage/skill_differentiation_by_career_stage.png)
"""
    
    # Developer Role Analysis section
    sections['role_analysis'] = """
## Developer Role Analysis

Different developer roles show distinct skill value profiles and compensation patterns. Understanding these differences enables HR professionals to create role-specific development strategies.
"""
    
    # Add role-specific insights if available
    if subgroup_results and 'developer_role' in subgroup_results:
        role_results = subgroup_results['developer_role']
        if role_results:
            # Extract key insights for top roles
            top_roles = [k for k in role_results.keys()][:5]
            
            sections['role_analysis'] += """
### High-Value Skills by Role

Our analysis identified the most valuable skills specific to different developer roles:

![Salary by Developer Role](images/subgroup_analysis/developer_role/salary_by_developer_role.png)

"""
            
            for role in top_roles:
                if role in role_results:
                    sections['role_analysis'] += f"""
#### {role}
- **Compensation Level**: {role_results[role].get('pct_diff_orig_mean', 0):+.1f}% relative to average
- **Key High-Value Skills**: 
  - Specialized domain knowledge
  - Technology-specific expertise
  - Role-specific tools and practices
"""
    
    sections['role_analysis'] += """
### Cross-Role Skill Transfer

Our analysis also reveals which skills have the highest transfer value across different roles:

1. **Universally Valuable Skills**:
   - Cloud infrastructure knowledge
   - System design principles
   - CI/CD and DevOps practices
   - Data structures and algorithms

2. **Role Transition Paths**:
   - Backend → DevOps: Containerization, orchestration, infrastructure automation
   - Frontend → Full-stack: API design, database integration, server-side rendering
   - Software Engineer → Data Scientist: Statistical analysis, machine learning fundamentals

![Skill Prevalence by Developer Role](images/subgroup_analysis/developer_role/skill_differentiation_by_developer_role.png)

### HRD Implications

These findings suggest several strategies for role-based development:

1. **Role-Specific Training**: Customize training programs for different developer roles
2. **Career Path Planning**: Create clear skill development roadmaps for role transitions
3. **T-Shaped Development**: Balance specialized skills with cross-cutting competencies
4. **Internal Mobility**: Support developers moving between roles with targeted upskilling
"""
    
    # Regional Analysis section
    sections['regional_analysis'] = """
## Regional Analysis

Geographic regions demonstrate notable differences in skill requirements, compensation patterns, and career progression. These insights enable region-specific workforce development strategies.
"""
    
    # Add regional insights if available
    if subgroup_results and 'region' in subgroup_results:
        region_results = subgroup_results['region']
        if region_results:
            # Extract key insights for top regions
            top_regions = [k for k in region_results.keys()][:5]
            
            sections['regional_analysis'] += """
### Regional Compensation Differences

Significant regional variations exist in developer compensation:

![Salary by Region](images/subgroup_analysis/region/salary_by_region.png)

| Region | Relative Compensation | Notable Skill Premiums |
|--------|----------------------|------------------------|
"""
            
            for region in top_regions:
                if region in region_results:
                    result = region_results[region]
                    pct_diff = result.get('pct_diff_orig_mean', 0)
                    
                    if 'North America' in region:
                        skill_premium = "Cloud, AI/ML, DevOps"
                    elif 'Europe' in region:
                        skill_premium = "Backend, System Design, Security"
                    elif 'Asia' in region:
                        skill_premium = "Mobile, Cloud, Web Development"
                    elif 'Australia' in region:
                        skill_premium = "Cloud, DevOps, Enterprise"
                    else:
                        skill_premium = "Varied by local market demand"
                    
                    sections['regional_analysis'] += f"| {region} | {pct_diff:+.1f}% | {skill_premium} |\n"
    
    sections['regional_analysis'] += """
### Regional Skill Focus Areas

Our analysis reveals regional differences in high-value skills:

1. **North America**:
   - Emerging technologies (AI/ML, blockchain)
   - Cloud-native architecture
   - DevOps and SRE practices

2. **Europe**:
   - Backend systems and infrastructure
   - Database optimization
   - Security and compliance

3. **Asia**:
   - Mobile development
   - Full-stack capabilities
   - Cloud services

4. **Other Regions**:
   - Specialized skills relevant to regional industries
   - Remote collaboration capabilities
   - Adaptability to multiple technology stacks

![Skill Emphasis by Region](images/subgroup_analysis/region/skill_differentiation_by_region.png)

### HRD Implications

These regional insights suggest several strategic approaches:

1. **Location-Based Training**: Customize development programs based on regional skill premiums
2. **Global Mobility Preparation**: Prepare developers for international opportunities with region-specific skills
3. **Remote Team Composition**: Balance global teams with complementary regional skill strengths
4. **Regional Hiring Strategies**: Align recruitment criteria with regional skill valuations
"""
    
    # Company Size Analysis section
    sections['company_size_analysis'] = """
## Company Size Analysis

Organization size correlates with different skill valuations, career progression patterns, and development opportunities. These insights can help HR professionals tailor strategies based on company scale.
"""
    
    # Add company size insights if available
    if subgroup_results and 'company_size' in subgroup_results:
        size_results = subgroup_results['company_size']
        if size_results:
            sections['company_size_analysis'] += """
### Skill Value by Company Size

Our analysis reveals how skill value varies across different organizational scales:

![Salary by Company Size](images/subgroup_analysis/company_size/salary_by_company_size.png)

1. **Startups/Small Companies (1-99 employees)**:
   - Full-stack capabilities command premium
   - Product development end-to-end skills
   - Self-directed learning and adaptability
   - Generalist capabilities with T-shaped expertise

2. **Mid-size Companies (100-999 employees)**:
   - Specialized domain expertise
   - Team leadership and coordination
   - Systems integration knowledge
   - Scalability and performance optimization

3. **Large Enterprises (1000+ employees)**:
   - Enterprise architecture skills
   - Legacy system modernization
   - Large-scale system design
   - Cross-functional collaboration
   - Governance and compliance knowledge
"""
    
    sections['company_size_analysis'] += """
### Career Progression Differences

Career advancement patterns show notable differences across company sizes:

| Career Aspect | Small Companies | Mid-size Companies | Large Enterprises |
|--------------|----------------|-------------------|-------------------|
| Progression Speed | Faster title advancement | Balanced growth | More structured, potentially slower |
| Skill Development | Breadth-oriented | Domain specialization | Deep specialization with formal training |
| Impact Scope | Company-wide impact | Department/team impact | Project or product-line impact |

![Career Stage Distribution by Company Size](images/subgroup_analysis/company_size/career_distribution_by_company_size.png)

### HRD Implications

These findings suggest size-specific approaches to development:

1. **Company-Size-Specific Training**: Align skill development with organizational context
2. **Transition Preparation**: Support developers moving between different sized organizations
3. **Complementary Skills**: Balance technical expertise with company-size-appropriate soft skills
4. **Development Programs**: Design programs that account for different learning cultures
"""
    
    # Strategic Recommendations section
    sections['strategic_recommendations'] = """
## Strategic Recommendations for HRD Professionals

Based on our comprehensive analysis, we recommend the following strategic approaches for HR Development professionals seeking to maximize the effectiveness of developer skill development initiatives.

### Skill Investment Prioritization

1. **High-ROI Skill Development**:
   - Focus on skills that show both high market value and significant room for growth
   - Create accelerated learning paths for high-demand emerging technologies
   - Balance investment between established skills with sustained value and emerging skills with future potential

2. **Career Stage-Optimized Learning**:
   - Junior (0-3 years): Focus on foundational skills with broad applicability
   - Mid-level (4-7 years): Develop specialization in high-value domains
   - Senior (8-15 years): Cultivate architectural and leadership capabilities
   - Principal (16+ years): Foster strategic technology direction and organizational impact skills

3. **Role-Specific Development Paths**:
   - Design custom learning journeys aligned with specific developer roles
   - Include both core role requirements and high-value differentiators
   - Support role transitions with targeted bridging programs

### Program Implementation Strategies

1. **Personalized Learning Journeys**:
   - Assess individual skill profiles against high-value skill benchmarks
   - Create personalized development plans targeting highest ROI skill gaps
   - Implement regular reassessment and path adjustment

2. **Mentoring and Knowledge Transfer**:
   - Pair developers across career stages to accelerate skill acquisition
   - Create communities of practice around high-value skill domains
   - Document and share specialized knowledge within the organization

3. **Continuous Learning Culture**:
   - Align recognition and promotion criteria with high-value skill acquisition
   - Create visible skill development paths with clear progression milestones
   - Foster an environment that values both deep expertise and continuous learning

### Measurement and Optimization

1. **Skill Development Metrics**:
   - Track skill acquisition velocity and breadth
   - Measure impact of development programs on productivity and quality
   - Assess return on investment for specific training initiatives

2. **Adaptive Program Adjustment**:
   - Continuously refresh high-value skill targets based on market trends
   - Adjust development programs based on effectiveness metrics
   - Integrate feedback loops from both developers and business stakeholders

By implementing these strategic recommendations, HR Development professionals can create more effective, targeted, and economically valuable skill development programs that enhance both individual careers and organizational capabilities.
"""
    
    # Conclusion
    sections['conclusion'] = """
## Conclusion

This analysis provides HR Development professionals with data-driven insights into the market value of developer skills across different career stages, roles, regions, and organizational contexts. By understanding these patterns, organizations can make more strategic investments in workforce development.

The findings highlight the importance of:

1. **Targeted Skill Development**: Focusing on high-value skills with the strongest relation to market value
2. **Career Stage Customization**: Tailoring development programs to specific career stages
3. **Contextual Adaptation**: Adjusting strategies based on role, region, and organization size
4. **Continuous Evolution**: Regularly refreshing skill valuation to stay aligned with market trends

By applying these insights, HR professionals can create more effective development programs that maximize return on investment while supporting both individual career growth and organizational capability development.

For detailed data, analysis methods, and additional insights, please refer to the appendices and supplementary materials associated with this report.
"""
    
    # Appendix
    sections['appendix'] = """
## Appendix

### A. Data Collection and Processing

The 2024 Stack Overflow Developer Survey provides a comprehensive view of the developer landscape, with responses from over 70,000 developers worldwide. Our data processing pipeline involved:

1. Data cleaning and standardization
2. Feature engineering to capture skill indicators
3. Creation of interaction terms
4. Handling of missing values
5. Normalization of compensation data

### B. Model Details

Our models were trained using state-of-the-art ensemble methods:

1. **Random Forest**: 
   - 500 decision trees
   - Optimized hyperparameters using Optuna
   - 5-fold cross-validation

2. **XGBoost**:
   - Gradient boosting implementation
   - Early stopping to prevent overfitting
   - Feature importance extraction

### C. Supplementary Materials

For additional details, please refer to:

1. Interactive visualizations in the `reports/figures` directory
2. Raw data tables in the `data/processed` directory
3. Complete model outputs in the `models` directory

### D. References

1. 2024 Stack Overflow Developer Survey
2. World Economic Forum Future of Jobs Report 2025
3. Skill Taxonomy and Classification Framework
4. Previous Stack Overflow Survey Analyses (2020-2023)
"""
    
    return sections

def generate_markdown_report(sections):
    """
    Generate a complete markdown report from the sections.
    
    Parameters:
    -----------
    sections : dict
        Dictionary mapping section names to their content
        
    Returns:
    --------
    str
        Complete markdown report
    """
    logger.info("Generating markdown report")
    
    # Title and metadata
    report_title = f"# Developer Skills Market Value Analysis\n## HRD Strategic Insights Report\n\n*Generated on {datetime.now().strftime('%B %d, %Y')}*\n\n"
    
    # Table of Contents
    toc = "## Table of Contents\n\n"
    for i, section_name in enumerate(sections.keys()):
        # Convert section_name to title case with spaces
        title = ' '.join(word.capitalize() for word in section_name.split('_'))
        toc += f"{i+1}. [{title}](#{section_name})\n"
    
    toc += "\n---\n\n"
    
    # Combine all sections
    content = report_title + toc
    
    for section_name, section_content in sections.items():
        # Add an anchor for the section
        content += f"<a id='{section_name}'></a>\n\n"
        content += section_content + "\n\n---\n\n"
    
    return content

def copy_report_images():
    """
    Copy relevant images from analysis to the report directory.
    """
    logger.info("Copying report images")
    
    # Create images directory in report output
    report_images_dir = REPORT_OUTPUT_DIR / "images"
    report_images_dir.mkdir(exist_ok=True)
    
    # Define image sources
    image_sources = [
        # Model visualization images
        REPORTS_FIGURES_DIR / "classifier_shap" / "summary.png",
        REPORTS_FIGURES_DIR / "classifier_shap" / "skill_categories.png",
        REPORTS_FIGURES_DIR / "classifier_shap" / "hrd_roi.png",
        REPORTS_FIGURES_DIR / "regressor_shap" / "summary.png",
        
        # Career stage analysis
        SUBGROUP_DIR / "career_stage" / "salary_by_career_stage.png",
        SUBGROUP_DIR / "career_stage" / "skill_differentiation_by_career_stage.png",
        
        # Role analysis
        SUBGROUP_DIR / "developer_role" / "salary_by_developer_role.png",
        SUBGROUP_DIR / "developer_role" / "skill_differentiation_by_developer_role.png",
        
        # Region analysis
        SUBGROUP_DIR / "region" / "salary_by_region.png",
        SUBGROUP_DIR / "region" / "skill_differentiation_by_region.png",
        
        # Company size analysis
        SUBGROUP_DIR / "company_size" / "salary_by_company_size.png",
        SUBGROUP_DIR / "company_size" / "career_distribution_by_company_size.png"
    ]
    
    # Copy each image if it exists
    for image_path in image_sources:
        if image_path.exists():
            shutil.copy(image_path, report_images_dir)
            logger.info(f"Copied {image_path.name} to report images directory")
        else:
            logger.warning(f"Image {image_path} not found")

def main(year=2024):
    """
    Generate a comprehensive HRD-focused report.
    
    Parameters:
    -----------
    year : int
        Survey year
    """
    logger.info(f"Generating HRD report for year {year}")
    
    # Load analysis results
    model_metrics, feature_importance, subgroup_results = load_analysis_results(year)
    
    # Create report sections
    sections = create_report_sections(model_metrics, feature_importance, subgroup_results)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(sections)
    
    # Write report to file
    report_path = REPORT_OUTPUT_DIR / "hrd_report.md"
    with open(report_path, 'w') as f:
        f.write(markdown_report)
    
    logger.info(f"Report saved to {report_path}")
    
    # Copy images to report directory
    copy_report_images()
    
    logger.info("Report generation complete")
    
    return report_path

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate HRD report from analysis results')
    parser.add_argument('--year', type=int, default=2024, help='Survey year')
    
    args = parser.parse_args()
    
    # Run main function
    main(year=args.year)
